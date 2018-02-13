$(document).ready(function(){
    $("#ask_button").click(function(){
        $.ajax({
          type: "POST",
          url: "http://127.0.0.1:5000/predict",
          dataType: "text",
          data:
          {
            paragraph: $('#paragraph').val(),
            question: $('#answer').val()
          },
          success: function(data) {
                    $("#answer").val(data);
          },
          error: function(response, textStatus) {
                $("#answer").val(textStatus);
          }
        });
    });
});

https://stackoverflow.com/questions/17909646/counting-and-limiting-words-in-a-textarea
$(document).ready(function() {
  $("#paragraph").on('keyup', function() {
    var words = this.value.match(/\S+/g).length;

    if (words > 200) {
      // Split the string on first 200 words and rejoin on spaces
      var trimmed = $(this).val().split(/\s+/, 200).join(" ");
      // Add a space at the end to make sure more typing creates new words
      $(this).val(trimmed + " ");
    }
    else {
      //$('#display_count').text(words);
      //$('#word_left').text(200-words);
    }
  });
});